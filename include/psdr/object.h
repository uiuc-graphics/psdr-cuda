#pragma once

#include <drjit/traversable_base.h>

// Object Base Class
NAMESPACE_BEGIN(psdr_jit)

class Object : public drjit::TraversableBase {
public:
    explicit Object(const char *domain = nullptr) : m_domain(domain) {}
    Object(const Object &other) : Object(other.m_domain) { m_id = other.m_id; }
    Object &operator=(const Object &other) { m_id = other.m_id; return *this; }
    virtual ~Object() { if (m_registered) jit_registry_remove(this); }

    // Register only scene-owned, configured objects. Symbolic calls can visit
    // every instance in a domain, including otherwise unused Python objects.
    void register_instance() {
        if (m_domain && !m_registered) {
            drjit::registry_put("", m_domain, this);
            m_registered = true;
        }
    }

    DR_TRAVERSE_CB(drjit::TraversableBase)

    virtual std::string type_name() const = 0;

    void log(const char *msg) const {
        std::cout << "[" << type_name() << "] " << msg << std::endl;
    }

    virtual std::string to_string() const {
        std::stringstream oss;
        oss << type_name();
        if ( m_id != "" ) oss << "[id=" << m_id << "]";
        return oss.str();
    }

    std::string m_id = "";
private:
    const char *m_domain;
    bool m_registered = false;
};

}
